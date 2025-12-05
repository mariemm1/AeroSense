import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SummaryRow } from './summary-row';

describe('SummaryRow', () => {
  let component: SummaryRow;
  let fixture: ComponentFixture<SummaryRow>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SummaryRow]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SummaryRow);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
